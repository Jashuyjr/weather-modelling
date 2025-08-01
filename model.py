import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load weather data
df = pd.read_csv('data/weather.csv', parse_dates=['Date'], index_col='Date')

# Visualize the time series
df['Temperature'].plot(title='Temperature Over Time')
plt.show()

# Train ARIMA model
model = ARIMA(df['Temperature'], order=(2,1,2))
model_fit = model.fit()

# Forecast the next 5 days
forecast = model_fit.forecast(steps=5)
print('Next 5 Days Temperature Forecast:\n', forecast)
