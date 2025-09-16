"""
Advanced Demand Forecasting Models
Includes multiple forecasting approaches: Linear Regression, ARIMA-style, and Seasonal models
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import sqlite3
import warnings
warnings.filterwarnings('ignore')

class DemandForecaster:
    def __init__(self, db_path='data/demand_forecast.db'):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        
    def load_data(self, product_id=None, category=None):
        """Load sales data from database"""
        conn = sqlite3.connect(self.db_path)
        
        if product_id:
            query = "SELECT * FROM sales WHERE product_id = ? ORDER BY date"
            df = pd.read_sql_query(query, conn, params=(product_id,))
        elif category:
            query = "SELECT * FROM sales WHERE category = ? ORDER BY date"
            df = pd.read_sql_query(query, conn, params=(category,))
        else:
            query = "SELECT * FROM sales ORDER BY date"
            df = pd.read_sql_query(query, conn)
            
        conn.close()
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def create_features(self, df, target_col='units_sold', lag_periods=[1, 2, 3, 7, 14, 30]):
        """Create lag features and time-based features"""
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        # Create lag features
        for lag in lag_periods:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Create rolling statistics
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        
        # Time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
        
        # Seasonal features
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def prepare_data(self, df, target_col='units_sold', test_size=0.2):
        """Prepare data for training"""
        # Create features
        df_features = self.create_features(df, target_col)
        
        # Remove rows with NaN values
        df_clean = df_features.dropna()
        
        # Select feature columns
        feature_cols = [col for col in df_clean.columns if col not in 
                       ['date', 'product_id', 'product_name', 'category', 'revenue', 'price', target_col]]
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        # Split data (time series - no shuffling)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train_linear_model(self, X_train, y_train, model_name='linear'):
        """Train linear regression model"""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Store model and scaler
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        
        return model
    
    def train_ridge_model(self, X_train, y_train, model_name='ridge', alpha=1.0):
        """Train Ridge regression model"""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        
        # Store model and scaler
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        
        return model
    
    def train_random_forest(self, X_train, y_train, model_name='rf', n_estimators=100):
        """Train Random Forest model"""
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        # Store model (no scaling needed for RF)
        self.models[model_name] = model
        self.scalers[model_name] = None
        
        return model
    
    def predict(self, X_test, model_name='linear'):
        """Make predictions using trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train the model first.")
        
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
            predictions = model.predict(X_test_scaled)
        else:
            predictions = model.predict(X_test)
        
        return predictions
    
    def evaluate_model(self, y_true, y_pred):
        """Evaluate model performance"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
    
    def forecast_future(self, df, model_name='linear', periods=30, target_col='units_sold'):
        """Forecast future demand"""
        # Prepare the last known data point
        df_features = self.create_features(df, target_col)
        df_clean = df_features.dropna()
        
        # Get feature columns
        feature_cols = [col for col in df_clean.columns if col not in 
                       ['date', 'product_id', 'product_name', 'category', 'revenue', 'price', target_col]]
        
        # Get the last row for forecasting
        last_row = df_clean.iloc[-1:][feature_cols]
        
        forecasts = []
        current_data = df_clean.copy()
        
        for i in range(periods):
            # Make prediction
            pred = self.predict(last_row, model_name)[0]
            forecasts.append(pred)
            
            # Update data for next prediction (simplified approach)
            # In practice, you'd want more sophisticated feature updating
            next_date = current_data['date'].iloc[-1] + pd.Timedelta(days=1)
            
            # Create next row (this is a simplified approach)
            next_row = last_row.copy()
            # Update lag features (shift previous predictions)
            if len(forecasts) >= 1:
                next_row.iloc[0, 0] = pred  # lag_1
            if len(forecasts) >= 2:
                next_row.iloc[0, 1] = forecasts[-2]  # lag_2
            if len(forecasts) >= 3:
                next_row.iloc[0, 2] = forecasts[-3]  # lag_3
            
            last_row = next_row
        
        return forecasts
    
    def get_feature_importance(self, model_name='rf', feature_cols=None):
        """Get feature importance for tree-based models"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            print(f"Model {model_name} does not have feature importance.")
            return None

class ProductDemandAnalyzer:
    def __init__(self, db_path='data/demand_forecast.db'):
        self.db_path = db_path
        
    def get_products(self):
        """Get list of all products"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM products", conn)
        conn.close()
        return df
    
    def get_sales_summary(self, product_id=None, category=None):
        """Get sales summary statistics"""
        conn = sqlite3.connect(self.db_path)
        
        if product_id:
            query = """
            SELECT 
                product_name,
                category,
                COUNT(*) as total_days,
                SUM(units_sold) as total_units,
                AVG(units_sold) as avg_daily_units,
                MIN(units_sold) as min_daily_units,
                MAX(units_sold) as max_daily_units,
                SUM(revenue) as total_revenue
            FROM sales 
            WHERE product_id = ?
            GROUP BY product_id, product_name, category
            """
            df = pd.read_sql_query(query, conn, params=(product_id,))
        elif category:
            query = """
            SELECT 
                category,
                COUNT(*) as total_days,
                SUM(units_sold) as total_units,
                AVG(units_sold) as avg_daily_units,
                MIN(units_sold) as min_daily_units,
                MAX(units_sold) as max_daily_units,
                SUM(revenue) as total_revenue
            FROM sales 
            WHERE category = ?
            GROUP BY category
            """
            df = pd.read_sql_query(query, conn, params=(category,))
        else:
            query = """
            SELECT 
                product_name,
                category,
                COUNT(*) as total_days,
                SUM(units_sold) as total_units,
                AVG(units_sold) as avg_daily_units,
                MIN(units_sold) as min_daily_units,
                MAX(units_sold) as max_daily_units,
                SUM(revenue) as total_revenue
            FROM sales 
            GROUP BY product_id, product_name, category
            ORDER BY total_revenue DESC
            """
            df = pd.read_sql_query(query, conn)
            
        conn.close()
        return df
