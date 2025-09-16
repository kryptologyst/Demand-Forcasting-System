# Demand Forecasting System

A comprehensive demand forecasting system that predicts future product demand based on historical sales data. This system is crucial for inventory management, logistics, and production planning.

## Features

- **Multiple Forecasting Models**: Linear Regression, Ridge Regression, and Random Forest
- **Web-based Dashboard**: Interactive UI for visualizing forecasts and analytics
- **Real-time Analytics**: Sales trends, category analysis, and performance metrics
- **Mock Database**: Realistic sales data with seasonal patterns and trends
- **API Endpoints**: RESTful API for custom forecasting requests
- **Feature Engineering**: Advanced lag features, rolling statistics, and seasonal components

## Screenshots

### Dashboard
![Dashboard](screenshots/dashboard.png)

### Product Forecast
![Product Forecast](screenshots/product_forecast.png)

### Analytics
![Analytics](screenshots/analytics.png)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/demand-forecasting-system.git
   cd demand-forecasting-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate mock database**
   ```bash
   python data/mock_database.py
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:5000`

## 📁 Project Structure

```
demand-forecasting-system/
├── app.py                      # Flask web application
├── requirements.txt            # Python dependencies
├── README.md                  # Project documentation
├── 0059.py                    # Original simple forecasting script
├── data/
│   ├── mock_database.py       # Mock database generator
│   └── demand_forecast.db     # SQLite database (generated)
├── models/
│   └── forecasting_models.py  # Advanced forecasting models
├── templates/
│   ├── base.html             # Base template
│   ├── index.html            # Dashboard
│   ├── forecast.html         # Product forecast page
│   ├── category_forecast.html # Category forecast page
│   ├── analytics.html        # Analytics dashboard
│   └── error.html            # Error page
└── static/
    ├── css/                  # Custom stylesheets
    └── js/                   # Custom JavaScript
```

## 🔧 Usage

### Web Interface

1. **Dashboard**: Overview of all products and key metrics
2. **Product Forecasting**: Individual product demand forecasts
3. **Category Analysis**: Aggregate forecasting by product category
4. **Analytics**: Sales trends and performance insights

### API Usage

```python
import requests

# Forecast for a specific product
response = requests.post('http://localhost:5000/api/forecast', json={
    'product_id': 1,
    'periods': 30,
    'model_type': 'rf'
})

forecast_data = response.json()
print(forecast_data['forecast'])
```

### Command Line Usage

```python
from models.forecasting_models import DemandForecaster

# Initialize forecaster
forecaster = DemandForecaster()

# Load data for a specific product
df = forecaster.load_data(product_id=1)

# Prepare data and train model
X_train, X_test, y_train, y_test, feature_cols = forecaster.prepare_data(df)
forecaster.train_random_forest(X_train, y_train, 'my_model')

# Generate forecast
forecast = forecaster.forecast_future(df, 'my_model', periods=30)
print(f"30-day forecast: {forecast}")
```

## Models

### 1. Linear Regression
- Simple baseline model using lag features
- Fast training and prediction
- Good interpretability

### 2. Ridge Regression
- Regularized linear model
- Handles multicollinearity better
- Prevents overfitting

### 3. Random Forest
- Ensemble method with multiple decision trees
- Captures non-linear patterns
- Provides feature importance

## Features Engineering

- **Lag Features**: Previous 1, 2, 3, 7, 14, and 30 days
- **Rolling Statistics**: Moving averages and standard deviations
- **Time Features**: Day of week, month, quarter, weekend indicator
- **Seasonal Features**: Sine/cosine transformations for cyclical patterns

## Database Schema

### Products Table
- `id`: Product identifier
- `name`: Product name
- `category`: Product category
- `price`: Product price

### Sales Table
- `date`: Sale date
- `product_id`: Product identifier
- `product_name`: Product name
- `category`: Product category
- `units_sold`: Number of units sold
- `revenue`: Total revenue
- `price`: Product price

## Model Evaluation

The system evaluates models using multiple metrics:

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

## Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
1. Set environment variables
2. Use a production WSGI server (e.g., Gunicorn)
3. Configure reverse proxy (e.g., Nginx)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Flask, scikit-learn, and Plotly
- Inspired by real-world demand forecasting challenges
- Uses Bootstrap for responsive UI design

## Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
- **GitHub**: [Your GitHub Profile](https://github.com/yourusername)

## Future Enhancements

- [ ] ARIMA and SARIMA models
- [ ] Prophet forecasting
- [ ] Real-time data integration
- [ ] Advanced visualization with D3.js
- [ ] Model deployment with Docker
- [ ] Automated model retraining
- [ ] Email alerts for forecast anomalies
- [ ] Export to Excel/PDF reports
# Demand-Forcasting-System
