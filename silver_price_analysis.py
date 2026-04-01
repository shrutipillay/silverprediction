import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Calculate the date 10 years ago
end_date = datetime.now()
start_date = end_date - timedelta(days=365*10)

print(f"Fetching silver price data from {start_date.date()} to {end_date.date()}...")

# Download silver price data (SI=F is silver futures)
silver_data = yf.download('SI=F', start=start_date, end=end_date, progress=False)

# Remove any NaN values
silver_data = silver_data.dropna()

# Get the Close prices (handle multi-level columns if they exist)
if isinstance(silver_data.columns, pd.MultiIndex):
    close_prices = silver_data['Close', 'SI=F']
else:
    close_prices = silver_data['Close']

# Display summary statistics
print("\nSilver Price Summary (Last 10 Years):")
print(f"Total records: {len(silver_data)}")
print(f"\nPrice Statistics:")
print(f"  Minimum: ${close_prices.min():.2f}")
print(f"  Maximum: ${close_prices.max():.2f}")
print(f"  Average: ${close_prices.mean():.2f}")
print(f"  Latest: ${close_prices.iloc[-1]:.2f}")

# Create the chart
plt.figure(figsize=(14, 7))
plt.plot(silver_data.index, close_prices, linewidth=1.5, color='#4169E1')
plt.title('Silver Price - Last 10 Years Daily Data', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD per Troy Ounce)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the chart
output_path = 'silver_price_chart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nChart saved as '{output_path}'")

# Save data to CSV
csv_path = 'silver_price_data.csv'
silver_data.to_csv(csv_path)
print(f"Data saved as '{csv_path}'")

# Display the chart
plt.show()
