"""
Flask Web Application for Demand Forecasting System
Provides a web interface for demand forecasting and analysis
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.utils
import json
import sqlite3
import os
from models.forecasting_models import DemandForecaster, ProductDemandAnalyzer
from data.mock_database import MockDatabaseGenerator

app = Flask(__name__)

# Initialize forecaster and analyzer
forecaster = DemandForecaster()
analyzer = ProductDemandAnalyzer()

@app.route('/')
def index():
    """Main dashboard page"""
    try:
        # Check if database exists, create if not
        if not os.path.exists('data/demand_forecast.db'):
            generator = MockDatabaseGenerator('data/demand_forecast.db')
            generator.create_database()
        
        # Get products and summary statistics
        products = analyzer.get_products()
        sales_summary = analyzer.get_sales_summary()
        
        return render_template('index.html', 
                             products=products.to_dict('records'),
                             sales_summary=sales_summary.to_dict('records'))
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/forecast/<int:product_id>')
def forecast_product(product_id):
    """Forecast demand for a specific product"""
    try:
        # Load product data
        df = forecaster.load_data(product_id=product_id)
        
        if df.empty:
            return render_template('error.html', error=f"No data found for product ID {product_id}")
        
        # Prepare data and train models
        X_train, X_test, y_train, y_test, feature_cols = forecaster.prepare_data(df)
        
        # Train multiple models
        forecaster.train_linear_model(X_train, y_train, 'linear')
        forecaster.train_ridge_model(X_train, y_train, 'ridge')
        forecaster.train_random_forest(X_train, y_train, 'rf')
        
        # Make predictions
        predictions = {}
        evaluations = {}
        
        for model_name in ['linear', 'ridge', 'rf']:
            pred = forecaster.predict(X_test, model_name)
            predictions[model_name] = pred
            evaluations[model_name] = forecaster.evaluate_model(y_test, pred)
        
        # Generate future forecasts
        future_forecasts = {}
        for model_name in ['linear', 'ridge', 'rf']:
            future_forecasts[model_name] = forecaster.forecast_future(df, model_name, periods=30)
        
        # Create plots
        plots = create_forecast_plots(df, X_test, y_test, predictions, future_forecasts)
        
        # Get product info
        product_info = analyzer.get_products()
        product_info = product_info[product_info['id'] == product_id].iloc[0]
        
        return render_template('forecast.html',
                             product_info=product_info.to_dict(),
                             evaluations=evaluations,
                             plots=plots,
                             feature_importance=get_feature_importance_data(forecaster, feature_cols))
        
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/category/<category>')
def forecast_category(category):
    """Forecast demand for a product category"""
    try:
        # Load category data (aggregated)
        df = forecaster.load_data(category=category)
        
        if df.empty:
            return render_template('error.html', error=f"No data found for category {category}")
        
        # Aggregate by date
        df_agg = df.groupby('date').agg({
            'units_sold': 'sum',
            'revenue': 'sum'
        }).reset_index()
        
        # Prepare data and train models
        X_train, X_test, y_train, y_test, feature_cols = forecaster.prepare_data(df_agg)
        
        # Train models
        forecaster.train_linear_model(X_train, y_train, f'{category}_linear')
        forecaster.train_ridge_model(X_train, y_train, f'{category}_ridge')
        forecaster.train_random_forest(X_train, y_train, f'{category}_rf')
        
        # Make predictions
        predictions = {}
        evaluations = {}
        
        for model_type in ['linear', 'ridge', 'rf']:
            model_name = f'{category}_{model_type}'
            pred = forecaster.predict(X_test, model_name)
            predictions[model_type] = pred
            evaluations[model_type] = forecaster.evaluate_model(y_test, pred)
        
        # Generate future forecasts
        future_forecasts = {}
        for model_type in ['linear', 'ridge', 'rf']:
            model_name = f'{category}_{model_type}'
            future_forecasts[model_type] = forecaster.forecast_future(df_agg, model_name, periods=30)
        
        # Create plots
        plots = create_forecast_plots(df_agg, X_test, y_test, predictions, future_forecasts)
        
        # Get category summary
        category_summary = analyzer.get_sales_summary(category=category)
        
        return render_template('category_forecast.html',
                             category=category,
                             category_summary=category_summary.to_dict('records')[0],
                             evaluations=evaluations,
                             plots=plots)
        
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/forecast', methods=['POST'])
def api_forecast():
    """API endpoint for custom forecasting"""
    try:
        data = request.json
        product_id = data.get('product_id')
        periods = data.get('periods', 30)
        model_type = data.get('model_type', 'rf')
        
        # Load and prepare data
        df = forecaster.load_data(product_id=product_id)
        X_train, X_test, y_train, y_test, feature_cols = forecaster.prepare_data(df)
        
        # Train model
        if model_type == 'linear':
            forecaster.train_linear_model(X_train, y_train, 'api_model')
        elif model_type == 'ridge':
            forecaster.train_ridge_model(X_train, y_train, 'api_model')
        else:
            forecaster.train_random_forest(X_train, y_train, 'api_model')
        
        # Generate forecast
        forecast = forecaster.forecast_future(df, 'api_model', periods=periods)
        
        # Evaluate model
        pred = forecaster.predict(X_test, 'api_model')
        evaluation = forecaster.evaluate_model(y_test, pred)
        
        return jsonify({
            'forecast': forecast,
            'evaluation': evaluation,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False})

@app.route('/analytics')
def analytics():
    """Analytics dashboard"""
    try:
        # Get overall statistics
        products = analyzer.get_products()
        sales_summary = analyzer.get_sales_summary()
        
        # Create analytics plots
        analytics_plots = create_analytics_plots()
        
        return render_template('analytics.html',
                             products=products.to_dict('records'),
                             sales_summary=sales_summary.to_dict('records'),
                             plots=analytics_plots)
        
    except Exception as e:
        return render_template('error.html', error=str(e))

def create_forecast_plots(df, X_test, y_test, predictions, future_forecasts):
    """Create plotly graphs for forecasting results"""
    plots = {}
    
    # Historical data plot
    fig_historical = go.Figure()
    fig_historical.add_trace(go.Scatter(
        x=df['date'],
        y=df['units_sold'],
        mode='lines',
        name='Historical Demand',
        line=dict(color='blue')
    ))
    fig_historical.update_layout(
        title='Historical Demand',
        xaxis_title='Date',
        yaxis_title='Units Sold',
        height=400
    )
    plots['historical'] = json.dumps(fig_historical, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Model comparison plot
    fig_comparison = go.Figure()
    
    # Get test dates (approximate)
    test_dates = df['date'].iloc[-len(y_test):].values
    
    fig_comparison.add_trace(go.Scatter(
        x=test_dates,
        y=y_test.values,
        mode='lines',
        name='Actual',
        line=dict(color='black', width=2)
    ))
    
    colors = {'linear': 'red', 'ridge': 'green', 'rf': 'orange'}
    for model_name, pred in predictions.items():
        fig_comparison.add_trace(go.Scatter(
            x=test_dates,
            y=pred,
            mode='lines',
            name=f'{model_name.upper()} Prediction',
            line=dict(color=colors[model_name], dash='dash')
        ))
    
    fig_comparison.update_layout(
        title='Model Predictions Comparison',
        xaxis_title='Date',
        yaxis_title='Units Sold',
        height=400
    )
    plots['comparison'] = json.dumps(fig_comparison, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Future forecast plot
    fig_future = go.Figure()
    
    # Last 60 days of historical data
    recent_data = df.tail(60)
    fig_future.add_trace(go.Scatter(
        x=recent_data['date'],
        y=recent_data['units_sold'],
        mode='lines',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Future dates
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
    
    for model_name, forecast in future_forecasts.items():
        fig_future.add_trace(go.Scatter(
            x=future_dates,
            y=forecast,
            mode='lines',
            name=f'{model_name.upper()} Forecast',
            line=dict(color=colors[model_name], dash='dot')
        ))
    
    fig_future.update_layout(
        title='30-Day Demand Forecast',
        xaxis_title='Date',
        yaxis_title='Units Sold',
        height=400
    )
    plots['future'] = json.dumps(fig_future, cls=plotly.utils.PlotlyJSONEncoder)
    
    return plots

def create_analytics_plots():
    """Create analytics plots"""
    plots = {}
    
    try:
        conn = sqlite3.connect('data/demand_forecast.db')
        
        # Sales by category over time
        query = """
        SELECT 
            date,
            category,
            SUM(units_sold) as total_units,
            SUM(revenue) as total_revenue
        FROM sales 
        GROUP BY date, category
        ORDER BY date
        """
        df_category = pd.read_sql_query(query, conn)
        df_category['date'] = pd.to_datetime(df_category['date'])
        
        fig_category = go.Figure()
        for category in df_category['category'].unique():
            cat_data = df_category[df_category['category'] == category]
            fig_category.add_trace(go.Scatter(
                x=cat_data['date'],
                y=cat_data['total_units'],
                mode='lines',
                name=category,
                stackgroup='one'
            ))
        
        fig_category.update_layout(
            title='Sales by Category Over Time',
            xaxis_title='Date',
            yaxis_title='Units Sold',
            height=400
        )
        plots['category_trends'] = json.dumps(fig_category, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Top products by revenue
        query = """
        SELECT 
            product_name,
            SUM(revenue) as total_revenue,
            SUM(units_sold) as total_units
        FROM sales 
        GROUP BY product_name
        ORDER BY total_revenue DESC
        LIMIT 10
        """
        df_top = pd.read_sql_query(query, conn)
        
        fig_top = go.Figure(data=[
            go.Bar(x=df_top['product_name'], y=df_top['total_revenue'])
        ])
        fig_top.update_layout(
            title='Top Products by Revenue',
            xaxis_title='Product',
            yaxis_title='Total Revenue ($)',
            height=400
        )
        plots['top_products'] = json.dumps(fig_top, cls=plotly.utils.PlotlyJSONEncoder)
        
        conn.close()
        
    except Exception as e:
        print(f"Error creating analytics plots: {e}")
    
    return plots

def get_feature_importance_data(forecaster, feature_cols):
    """Get feature importance for Random Forest model"""
    try:
        importance_df = forecaster.get_feature_importance('rf', feature_cols)
        if importance_df is not None:
            return importance_df.head(10).to_dict('records')
    except:
        pass
    return []

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
