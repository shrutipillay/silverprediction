import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Silver Price Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🥈 Silver Price Analysis & Forecasting Dashboard")
st.markdown("Real-time silver price data with Facebook Prophet forecasting")
st.markdown("---")

# Sidebar
st.sidebar.title("Dashboard Controls")
show_historical = st.sidebar.checkbox("📊 Show Historical Data", value=True)
show_test_predictions = st.sidebar.checkbox("🔴 Show Test Predictions", value=True)
show_forecast = st.sidebar.checkbox("🟠 Show 1-Year Forecast", value=True)
refresh_data = st.sidebar.button("🔄 Refresh Data")

@st.cache_data(ttl=3600)
def fetch_silver_data():
    """Fetch silver price data from Yahoo Finance"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)
    
    silver_data = yf.download('SI=F', start=start_date, end=end_date, progress=False)
    silver_data = silver_data.dropna()
    
    if isinstance(silver_data.columns, pd.MultiIndex):
        close_prices = silver_data['Close', 'SI=F'].values
    else:
        close_prices = silver_data['Close'].values
    
    dates = silver_data.index.to_pydatetime()
    return dates, close_prices

def train_model(dates, prices):
    """Train optimized Prophet model and forecast"""
    df = pd.DataFrame({'ds': dates, 'y': prices})
    
    # Remove outliers using IQR method
    Q1 = df['y'].quantile(0.25)
    Q3 = df['y'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['y'] >= lower_bound) & (df['y'] <= upper_bound)].copy()
    
    # Split data
    total_days = len(df)
    train_days = int(total_days - (365 * 0.5))
    
    train_df = df.iloc[:train_days].copy()
    test_df = df.iloc[train_days:].copy()
    
    # Scale training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = train_df.copy()
    train_scaled['y'] = scaler.fit_transform(train_df[['y']])
    
    # Fit optimized model (Config 2: Moderate)
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=5,
        seasonality_mode='multiplicative'
    )
    model.fit(train_scaled)
    
    # Predict on test set
    future_test = test_df[['ds']].copy()
    forecast_test_scaled = model.predict(future_test)
    forecast_test_values = scaler.inverse_transform(forecast_test_scaled[['yhat']].values)
    forecast_test_df = pd.DataFrame({
        'ds': forecast_test_scaled['ds'].values,
        'yhat': forecast_test_values.flatten(),
        'yhat_lower': scaler.inverse_transform(forecast_test_scaled[['yhat_lower']].values).flatten(),
        'yhat_upper': scaler.inverse_transform(forecast_test_scaled[['yhat_upper']].values).flatten()
    })
    
    # Future forecast (next year) with value clamping
    future_dates = pd.date_range(start=df.iloc[-1]['ds'], periods=365, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})
    forecast_future_scaled = model.predict(future_df)
    
    forecast_future_values_raw = scaler.inverse_transform(forecast_future_scaled[['yhat']].values).flatten()
    forecast_future_lower_raw = scaler.inverse_transform(forecast_future_scaled[['yhat_lower']].values).flatten()
    forecast_future_upper_raw = scaler.inverse_transform(forecast_future_scaled[['yhat_upper']].values).flatten()
    
    # Clamp to realistic range
    forecast_future = pd.DataFrame({
        'ds': forecast_future_scaled['ds'].values,
        'yhat': np.clip(forecast_future_values_raw, 1, 150),
        'yhat_lower': np.clip(forecast_future_lower_raw, 1, 150),
        'yhat_upper': np.clip(forecast_future_upper_raw, 1, 150)
    })
    
    # Calculate metrics
    actual_test = test_df['y'].values
    predicted_test = forecast_test_df['yhat'].values
    rmse = np.sqrt(mean_squared_error(actual_test, predicted_test))
    mape = np.mean(np.abs((actual_test - predicted_test) / actual_test)) * 100
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'forecast_test_df': forecast_test_df,
        'forecast_future': forecast_future,
        'rmse': rmse,
        'mape': mape
    }

# Main execution
try:
    # Load data
    with st.spinner('Loading silver price data...'):
        dates, prices = fetch_silver_data()
    
    # Display current metrics
    current_price = prices[-1]
    avg_price = np.mean(prices)
    max_price = np.max(prices)
    min_price = np.min(prices)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Current Price", f"${current_price:.2f}")
    with col2:
        st.metric("📊 10-Year Average", f"${avg_price:.2f}")
    with col3:
        st.metric("📈 10-Year High", f"${max_price:.2f}")
    with col4:
        st.metric("📉 10-Year Low", f"${min_price:.2f}")
    
    st.markdown("---")
    
    # Train model
    with st.spinner('Training Prophet model... (This takes ~30 seconds)'):
        results = train_model(dates, prices)
    
    # Display model metrics
    st.subheader("📈 Model Performance")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("RMSE", f"${results['rmse']:.2f}")
    with metric_col2:
        st.metric("MAPE", f"{results['mape']:.2f}%")
    with metric_col3:
        st.metric("Train Period", "~9.25 Years")
    with metric_col4:
        st.metric("Test Period", "6 Months")
    
    st.markdown("---")
    
    # Create visualizations
    st.subheader("📊 Price Analysis & Forecast")
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Train vs Test
    ax1 = axes[0]
    if show_historical:
        ax1.plot(results['train_df']['ds'], results['train_df']['y'], 
                label='Training Data', color='blue', linewidth=1.5, alpha=0.8)
    if show_test_predictions:
        ax1.plot(results['test_df']['ds'], results['test_df']['y'], 
                label='Test Data (Actual)', color='green', linewidth=1.5, alpha=0.8)
        ax1.plot(results['forecast_test_df']['ds'], results['forecast_test_df']['yhat'],
                label='Test Predictions', color='red', linewidth=1.5, linestyle='--', alpha=0.8)
        ax1.fill_between(results['forecast_test_df']['ds'],
                         results['forecast_test_df']['yhat_lower'],
                         results['forecast_test_df']['yhat_upper'],
                         alpha=0.2, color='red')
    
    ax1.set_title('Training & Test Data with Predictions', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price (USD/oz)', fontsize=10)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Complete forecast
    ax2 = axes[1]
    if show_historical:
        ax2.plot(results['train_df']['ds'], results['train_df']['y'],
                label='Historical Data', color='blue', linewidth=1, alpha=0.6)
    if show_test_predictions:
        ax2.plot(results['test_df']['ds'], results['test_df']['y'],
                color='green', linewidth=1.5, alpha=0.8)
        ax2.plot(results['forecast_test_df']['ds'], results['forecast_test_df']['yhat'],
                label='Test Predictions', color='red', linewidth=1.5, linestyle='--')
    if show_forecast:
        ax2.plot(results['forecast_future']['ds'], results['forecast_future']['yhat'],
                label='1-Year Forecast', color='orange', linewidth=2)
        ax2.fill_between(results['forecast_future']['ds'],
                        results['forecast_future']['yhat_lower'],
                        results['forecast_future']['yhat_upper'],
                        alpha=0.2, color='orange', label='95% Confidence Interval')
    
    ax2.axvline(x=results['test_df']['ds'].iloc[0], color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=results['train_df']['ds'].iloc[-1], color='purple', linestyle=':', alpha=0.5)
    ax2.set_title('Complete Historical Data, Test Predictions & 1-Year Forecast', 
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Price (USD/oz)', fontsize=10)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Forecast details
    st.subheader("🎯 1-Year Forecast Details")
    forecast_col1, forecast_col2, forecast_col3, forecast_col4 = st.columns(4)
    
    with forecast_col1:
        st.metric("Forecast Start", results['forecast_future']['ds'].iloc[0].strftime('%Y-%m-%d'))
    with forecast_col2:
        st.metric("Forecast End", results['forecast_future']['ds'].iloc[-1].strftime('%Y-%m-%d'))
    with forecast_col3:
        st.metric("Predicted Min", f"${results['forecast_future']['yhat'].min():.2f}")
    with forecast_col4:
        st.metric("Predicted Max", f"${results['forecast_future']['yhat'].max():.2f}")
    
    st.markdown("---")
    
    # Data tables
    if st.checkbox("📋 Show Data Tables"):
        st.write("**Last 10 Days of 1-Year Forecast**")
        forecast_display = results['forecast_future'].tail(10)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_display.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
        st.dataframe(forecast_display, use_container_width=True)
        
        st.write("**Last 10 Days of Test Predictions**")
        test_display = pd.DataFrame({
            'Date': results['forecast_test_df']['ds'].values[-10:],
            'Actual': results['test_df']['y'].values[-10:],
            'Predicted': results['forecast_test_df']['yhat'].values[-10:],
            'Difference': (results['test_df']['y'].values[-10:] - results['forecast_test_df']['yhat'].values[-10:])
        })
        st.dataframe(test_display, use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    **Dashboard Information:**
    - **Data Source:** Yahoo Finance (SI=F - Silver Futures)
    - **Model:** Facebook Prophet with MinMax Scaling
    - **Training Data:** ~9.25 years
    - **Test Data:** Last 6 months
    - **Forecast:** Next 12 months with 95% confidence interval
    - **Last Updated:** {} UTC
    """.format(datetime.now().strftime('%B %d, %Y at %H:%M')))

except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
    st.info("Please refresh the page or contact support if the issue persists.")
