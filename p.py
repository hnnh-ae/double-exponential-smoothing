import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Sample data
data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=20, freq='M'),
    'Value': [35, 39, 43, 48, 54, 59, 67, 73, 80, 87, 94, 101, 109, 117, 126, 135, 144, 155, 165, 176]
})

# Fit a double exponential smoothing model (Holt's method)
model = ExponentialSmoothing(data['Value'], trend='add', seasonal=None)
result = model.fit()

# Generate forecasts
forecast = result.forecast(steps=12)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Value'], label='Actual', marker='o')
plt.plot(data['Date'][-1] + pd.DateOffset(months=1) + pd.date_range(start=0, periods=12, freq='M'), forecast, label='Forecast', marker='o')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.title('Double Exponential Smoothing (Holt\'s Method) Forecast')
plt.show()

# Display the forecasted values
forecast
