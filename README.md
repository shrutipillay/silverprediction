# Silver Price Prediction using Facebook Prophet

A machine learning project for predicting silver prices using Facebook's Prophet forecasting library with time series analysis and MinMax scaling preprocessing.

## 📊 Project Overview

This project analyzes 10 years of daily silver price data from Yahoo Finance and uses Facebook Prophet to forecast prices with the following configuration:
- **Training Data:** 4.5 years (9.25 years of total historical data)
- **Test Data:** 6 months
- **Forecast Horizon:** 1 year ahead
- **Preprocessing:** MinMax Scaler (0-1 normalization)

## 📈 Key Results

### Model Performance
- **RMSE:** $29.04
- **MAPE:** 31.74%
- **Training Period:** April 4, 2016 - July 10, 2025
- **Test Period:** July 11, 2025 - April 1, 2026

### Price Statistics (10 Years)
- **Minimum:** $11.73 per troy ounce
- **Maximum:** $115.08 per troy ounce
- **Average:** $24.22 per troy ounce
- **Current:** $75.79 per troy ounce

### 1-Year Forecast (April 1, 2026 - March 31, 2027)
- **Predicted Range:** $37.53 - $45.60
- **Average Predicted:** $42.13
- **Final Prediction:** $45.60

## 🛠️ Technology Stack

- **Python 3.14.3**
- **Libraries:**
  - `yfinance` - Yahoo Finance data download
  - `pandas` - Data manipulation
  - `numpy` - Numerical computations
  - `matplotlib` - Visualization
  - `prophet` - Facebook's time series forecasting
  - `scikit-learn` - MinMax scaling and metrics
  - `streamlit` - Interactive web dashboard

## 📁 Project Files

1. **silver_price_analysis.py** - Downloads and visualizes 10 years of silver price data
2. **silver_prophet_forecast.py** - Trains Prophet model with preprocessing and generates forecasts
3. **streamlit_app.py** - Interactive Streamlit dashboard for visualization and exploration
4. **.gitignore** - Git ignore configuration

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/shrutipillay/silverprediction.git
cd silverprediction
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Linux/Mac
```

### Step 3: Install Dependencies
```bash
pip install yfinance pandas numpy matplotlib prophet scikit-learn streamlit
```

## 💻 Usage

### Option 1: Run Streamlit Dashboard (Recommended)
```bash
streamlit run streamlit_app.py
```
The dashboard will open at `http://localhost:8501`

### Option 2: Generate Forecast
```bash
python silver_prophet_forecast.py
```
Generates:
- `silver_forecast_prophet.png` - Visualization chart
- `silver_forecast_next_year.csv` - Forecast data
- `silver_test_predictions.csv` - Test set predictions

### Option 3: Download Historical Data
```bash
python silver_price_analysis.py
```
Generates:
- `silver_price_chart.png` - 10-year historical chart
- `silver_price_data.csv` - Historical price data

## 📊 Streamlit Dashboard Features

The interactive dashboard includes:
- ✅ Real-time silver price metrics
- ✅ Model performance metrics (RMSE, MAPE)
- ✅ Training & test data visualization
- ✅ 1-year forecast with confidence intervals
- ✅ Interactive toggles to show/hide chart elements
- ✅ Raw data tables for detailed analysis
- ✅ Responsive design

## 🔍 Model Details

### Prophet Configuration
- **Yearly Seasonality:** Enabled
- **Weekly Seasonality:** Enabled
- **Daily Seasonality:** Disabled
- **Confidence Interval:** 95%
- **Changepoint Prior Scale:** 0.05

### Data Preprocessing
- MinMax scaling to normalize data between 0 and 1
- Enables better convergence and interpretability
- Inverse transformation applied to predictions for original scale

## 📊 Output Files

### Visualizations
- `silver_price_chart.png` - 10-year historical price chart
- `silver_forecast_prophet.png` - Training, test, and forecast visualization

### Data Files
- `silver_price_data.csv` - 10 years of historical data
- `silver_forecast_next_year.csv` - 1-year forecast with confidence intervals
- `silver_test_predictions.csv` - Test set predictions vs. actuals

## 📈 Performance Metrics Explained

- **RMSE (Root Mean Square Error):** $29.04 - Average prediction error
- **MAPE (Mean Absolute Percentage Error):** 31.74% - Percentage error metric
- Both metrics indicate the model's accuracy on the test set

## 🎯 Future Improvements

- Implement LSTM/RNN models for comparison
- Add exogenous variables (USD strength, inflation)
- Ensemble methods combining multiple models
- Real-time data updates
- More granular seasonality analysis

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Shruti Pillay**
- GitHub: [@shrutipillay](https://github.com/shrutipillay)
- Email: pillay.shruti1@gmail.com

## 🔗 Repository

[Silver Prediction on GitHub](https://github.com/shrutipillay/silverprediction)

## 📚 Data Source

Historical data sourced from Yahoo Finance using the `SI=F` ticker (Silver Futures).

---

**Last Updated:** April 1, 2026
