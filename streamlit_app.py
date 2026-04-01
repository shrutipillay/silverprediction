import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Silver Price Forecasting", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🥈 Silver Price Analysis & Forecasting Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.title("Dashboard Controls")
show_historical = st.sidebar.checkbox("Show Historical Data", value=True)
show_test_predictions = st.sidebar.checkbox("Show Test Predictions", value=True)
show_forecast = st.sidebar.checkbox("Show 1-Year Forecast", value=True)

# Main content
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💰 Current Price", "$75.79")
with col2:
    st.metric("📊 10-Year Average", "$24.22")
with col3:
    st.metric("🎯 10-Year High", "$115.08")

st.markdown("---")

# Load data
@st.cache_data
def load_silver_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)
    silver_data = yf.download('SI=F', start=start_date, end=end_date, progress=False)
    silver_data = silver_data.dropna()
    
    if isinstance(silver_data.columns, pd.MultiIndex):
        close_prices = silver_data['Close', 'SI=F'].values
    else:
        close_prices = silver_data['Close'].values
    
    dates = silver_data.index
    df = pd.DataFrame({'ds': dates, 'y': close_prices})
    return df

@st.cache_data
def train_prophet_model(df):
    total_days = len(df)
    train_days = int(total_days - (365 * 0.5))
    test_start_idx = train_days
    
    train_df = df.iloc[:train_days].copy()
    test_df = df.iloc[test_start_idx:].copy()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = train_df.copy()
    train_scaled['y'] = scaler.fit_transform(train_df[['y']])
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
        changepoint_prior_scale=0.05
    )
    model.fit(train_scaled)
    
    # Test predictions
    future_test = test_df[['ds']].copy()
    forecast_test_scaled = model.predict(future_test)
    forecast_test_values = scaler.inverse_transform(forecast_test_scaled[['yhat']].values)
    forecast_test_df = pd.DataFrame({
        'ds': forecast_test_scaled['ds'],
        'yhat': forecast_test_values.flatten(),
        'yhat_lower': scaler.inverse_transform(forecast_test_scaled[['yhat_lower']].values).flatten(),
        'yhat_upper': scaler.inverse_transform(forecast_test_scaled[['yhat_upper']].values).flatten()
    })
    
    # Future predictions
    future_dates = pd.date_range(start=df.iloc[-1]['ds'], periods=365, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})
    forecast_scaled = model.predict(future_df)
    forecast_values = scaler.inverse_transform(forecast_scaled[['yhat']].values)
    forecast_future = pd.DataFrame({
        'ds': forecast_scaled['ds'],
        'yhat': forecast_values.flatten(),
        'yhat_lower': scaler.inverse_transform(forecast_scaled[['yhat_lower']].values).flatten(),
        'yhat_upper': scaler.inverse_transform(forecast_scaled[['yhat_upper']].values).flatten()
    })
    
    # Calculate metrics
    actual_test = test_df['y'].values
    predicted_test = forecast_test_df['yhat'].values
    rmse = np.sqrt(mean_squared_error(actual_test, predicted_test))
    mape = np.mean(np.abs((actual_test - predicted_test) / actual_test)) * 100
    
    return train_df, test_df, forecast_test_df, forecast_future, rmse, mape, scaler

# Load and process data
df = load_silver_data()
train_df, test_df, forecast_test_df, forecast_future, rmse, mape, scaler = train_prophet_model(df)

# Display metrics
st.subheader("📈 Model Performance Metrics")
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric("RMSE", f"${rmse:.2f}")
with metric_col2:
    st.metric("MAPE", f"{mape:.2f}%")
with metric_col3:
    st.metric("Training Period", "4.5 Years")
with metric_col4:
    st.metric("Test Period", "6 Months")

st.markdown("---")

# Visualizations
st.subheader("📊 Silver Price Analysis Charts")

# Create figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Plot 1: Training & Test with predictions
ax1 = axes[0]
if show_historical:
    ax1.plot(train_df['ds'], train_df['y'], label='Training Data (4.5 years)', color='blue', linewidth=1.5)
if show_test_predictions:
    ax1.plot(test_df['ds'], test_df['y'], label='Test Data (6 months - Actual)', color='green', linewidth=1.5)
    ax1.plot(forecast_test_df['ds'], forecast_test_df['yhat'], label='Test Predictions', 
             color='red', linewidth=1.5, linestyle='--')
    ax1.fill_between(forecast_test_df['ds'], forecast_test_df['yhat_lower'], 
                      forecast_test_df['yhat_upper'], alpha=0.2, color='red')

ax1.set_title('Silver Price: Training & Test Data with Predictions', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Price (USD per Troy Ounce)', fontsize=11)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Complete forecast
ax2 = axes[1]
if show_historical:
    ax2.plot(train_df['ds'], train_df['y'], label='Historical Data', color='blue', linewidth=1)
if show_test_predictions:
    ax2.plot(test_df['ds'], test_df['y'], color='green', linewidth=1.5)
    ax2.plot(forecast_test_df['ds'], forecast_test_df['yhat'], label='Test Predictions', 
             color='red', linewidth=1.5, linestyle='--')
if show_forecast:
    ax2.plot(forecast_future['ds'], forecast_future['yhat'], label='1-Year Forecast', 
             color='orange', linewidth=2)
    ax2.fill_between(forecast_future['ds'], forecast_future['yhat_lower'], 
                      forecast_future['yhat_upper'], alpha=0.2, color='orange', label='95% Confidence Interval')

ax2.axvline(x=test_df['ds'].iloc[0], color='gray', linestyle=':', alpha=0.7, label='Test Set Start')
ax2.axvline(x=df.iloc[-1]['ds'], color='purple', linestyle=':', alpha=0.7, label='Forecast Start')
ax2.set_title('Silver Price: Complete Historical Data, Test Predictions & 1-Year Forecast', 
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=11)
ax2.set_ylabel('Price (USD per Troy Ounce)', fontsize=11)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# Forecast details
st.subheader("🎯 1-Year Forecast Details")

forecast_col1, forecast_col2, forecast_col3, forecast_col4 = st.columns(4)

with forecast_col1:
    st.metric("Forecast Start", forecast_future.iloc[0]['ds'].strftime('%Y-%m-%d'))
with forecast_col2:
    st.metric("Forecast End", forecast_future.iloc[-1]['ds'].strftime('%Y-%m-%d'))
with forecast_col3:
    st.metric("Predicted Min", f"${forecast_future['yhat'].min():.2f}")
with forecast_col4:
    st.metric("Predicted Max", f"${forecast_future['yhat'].max():.2f}")

st.markdown("---")

# Data tables
if st.checkbox("Show Raw Data Tables"):
    st.subheader("📋 Forecast Data")
    st.write("**1-Year Forecast Data**")
    st.dataframe(forecast_future.tail(10), use_container_width=True)
    
    st.write("**Test Set Predictions** (Last 10 days)")
    test_results = pd.DataFrame({
        'Date': forecast_test_df['ds'],
        'Actual': test_df['y'].values,
        'Predicted': forecast_test_df['yhat'].values,
        'Error': test_df['y'].values - forecast_test_df['yhat'].values
    })
    st.dataframe(test_results.tail(10), use_container_width=True)

st.markdown("---")

# Footer
st.markdown("""
**Dashboard Information:**
- Data Source: Yahoo Finance (SI=F - Silver Futures)
- Model: Facebook Prophet with MinMax Scaling
- Training Period: Last 9.25 years of data
- Test Period: Last 6 months
- Forecast Horizon: Next 12 months
- Last Updated: April 1, 2026
""")
