import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Download data
print("Downloading silver price data...")
end_date = datetime.now()
start_date = end_date - timedelta(days=365*10)
silver_data = yf.download('SI=F', start=start_date, end=end_date, progress=False)
silver_data = silver_data.dropna()

# Get close prices
if isinstance(silver_data.columns, pd.MultiIndex):
    close_prices = silver_data['Close', 'SI=F'].values
else:
    close_prices = silver_data['Close'].values

dates = silver_data.index

# Create DataFrame in Prophet format
df = pd.DataFrame({
    'ds': dates,
    'y': close_prices
})

# Split data: 4.5 years train, 6 months test
total_days = len(df)
train_days = int(total_days - (365 * 0.5))  # Keep last 6 months for test
test_start_idx = train_days

print(f"\nData Split:")
print(f"Total records: {total_days}")
print(f"Training set: {train_days} records (from {df.iloc[0]['ds'].date()} to {df.iloc[train_days-1]['ds'].date()})")
print(f"Test set: {total_days - train_days} records (from {df.iloc[test_start_idx]['ds'].date()} to {df.iloc[-1]['ds'].date()})")

# Split data
train_df = df.iloc[:train_days].copy()
test_df = df.iloc[test_start_idx:].copy()

# Preprocessing: Scale the training data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = train_df.copy()
train_scaled['y'] = scaler.fit_transform(train_df[['y']])

# Train Prophet on scaled training data
print("\nTraining Prophet model on scaled data...")
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    interval_width=0.95,
    changepoint_prior_scale=0.05
)
model.fit(train_scaled)

# Make predictions on test set
print("Making predictions on test set...")
future_test = test_df[['ds']].copy()
forecast_test_scaled = model.predict(future_test)

# Inverse transform predictions to original scale
forecast_test_values = scaler.inverse_transform(
    forecast_test_scaled[['yhat']].values
)
forecast_test_df = pd.DataFrame({
    'ds': forecast_test_scaled['ds'],
    'yhat': forecast_test_values.flatten(),
    'yhat_lower': scaler.inverse_transform(
        forecast_test_scaled[['yhat_lower']].values
    ).flatten(),
    'yhat_upper': scaler.inverse_transform(
        forecast_test_scaled[['yhat_upper']].values
    ).flatten()
})

# Calculate RMSE on test set
actual_test = test_df['y'].values
predicted_test = forecast_test_df['yhat'].values
rmse = np.sqrt(mean_squared_error(actual_test, predicted_test))
mape = np.mean(np.abs((actual_test - predicted_test) / actual_test)) * 100

print(f"\nTest Set Performance:")
print(f"RMSE: ${rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Min actual: ${actual_test.min():.2f}, Predicted: ${predicted_test.min():.2f}")
print(f"Max actual: ${actual_test.max():.2f}, Predicted: ${predicted_test.max():.2f}")

# Predict for next 1 year
print("\nForecasting for next 1 year...")
future_dates = pd.date_range(start=df.iloc[-1]['ds'], periods=365, freq='D')
future_df = pd.DataFrame({'ds': future_dates})
forecast_scaled = model.predict(future_df)

# Inverse transform future predictions
forecast_values = scaler.inverse_transform(
    forecast_scaled[['yhat']].values
)
forecast_future = pd.DataFrame({
    'ds': forecast_scaled['ds'],
    'yhat': forecast_values.flatten(),
    'yhat_lower': scaler.inverse_transform(
        forecast_scaled[['yhat_lower']].values
    ).flatten(),
    'yhat_upper': scaler.inverse_transform(
        forecast_scaled[['yhat_upper']].values
    ).flatten()
})

# Create comprehensive visualization
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Plot 1: Full historical data with train/test split
ax1 = axes[0]
ax1.plot(train_df['ds'], train_df['y'], label='Training Data', color='blue', linewidth=1.5)
ax1.plot(test_df['ds'], test_df['y'], label='Test Data (Actual)', color='green', linewidth=1.5)
ax1.plot(forecast_test_df['ds'], forecast_test_df['yhat'], label='Test Predictions', 
         color='red', linewidth=1.5, linestyle='--')
ax1.fill_between(forecast_test_df['ds'], forecast_test_df['yhat_lower'], 
                  forecast_test_df['yhat_upper'], alpha=0.2, color='red')
ax1.set_title('Silver Price: Training & Test Data with Predictions', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Price (USD per Troy Ounce)', fontsize=11)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Complete forecast including next year
ax2 = axes[1]
ax2.plot(train_df['ds'], train_df['y'], label='Historical Data', color='blue', linewidth=1)
ax2.plot(test_df['ds'], test_df['y'], color='green', linewidth=1.5)
ax2.plot(forecast_test_df['ds'], forecast_test_df['yhat'], label='Test Predictions', 
         color='red', linewidth=1.5, linestyle='--')
ax2.plot(forecast_future['ds'], forecast_future['yhat'], label='1-Year Forecast', 
         color='orange', linewidth=2)
ax2.fill_between(forecast_future['ds'], forecast_future['yhat_lower'], 
                  forecast_future['yhat_upper'], alpha=0.2, color='orange', label='95% Confidence Interval')
ax2.axvline(x=test_df['ds'].iloc[0], color='gray', linestyle=':', alpha=0.7, label='Test Set Start')
ax2.axvline(x=df.iloc[-1]['ds'], color='purple', linestyle=':', alpha=0.7, label='Forecast Start')
ax2.set_title('Silver Price: Historical Data, Test Predictions, and 1-Year Forecast', 
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=11)
ax2.set_ylabel('Price (USD per Troy Ounce)', fontsize=11)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('silver_forecast_prophet.png', dpi=300, bbox_inches='tight')
print("Chart saved as 'silver_forecast_prophet.png'")

# Save forecast data
forecast_future.to_csv('silver_forecast_next_year.csv', index=False)
test_results = pd.DataFrame({
    'Date': forecast_test_df['ds'],
    'Actual': actual_test,
    'Predicted': predicted_test,
    'Residual': actual_test - predicted_test
})
test_results.to_csv('silver_test_predictions.csv', index=False)
print("Forecast data saved as 'silver_forecast_next_year.csv'")
print("Test predictions saved as 'silver_test_predictions.csv'")

# Print forecast summary
print("\n1-Year Forecast Summary:")
print(f"Forecast Start: {forecast_future.iloc[0]['ds'].date()}")
print(f"Forecast End: {forecast_future.iloc[-1]['ds'].date()}")
print(f"Predicted Min: ${forecast_future['yhat'].min():.2f}")
print(f"Predicted Max: ${forecast_future['yhat'].max():.2f}")
print(f"Predicted Avg: ${forecast_future['yhat'].mean():.2f}")
print(f"Final Prediction (1 year ahead): ${forecast_future.iloc[-1]['yhat']:.2f}")

plt.show()
