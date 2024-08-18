import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your real sales data from a CSV file (replace 'data/sales_data.csv' with your file path)
df = pd.read_csv('data/sales_data.csv')

# Ensure that 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Resample data to a daily frequency (if not already)
df = df.resample('D').sum()

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train_data, test_data = df[:train_size], df[train_size:]

# Fit a SARIMA model
order = (1, 1, 1)  # (p, d, q)
seasonal_order = (1, 1, 1, 7)  # (P, D, Q, S) - assuming weekly seasonality
model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
results = model.fit()

# Make predictions on the test set
y_pred = results.get_forecast(steps=len(test_data))
y_pred_mean = y_pred.predicted_mean

# Calculate forecast accuracy metrics
mae = mean_absolute_error(test_data, y_pred_mean)
mse = mean_squared_error(test_data, y_pred_mean)
rmse = np.sqrt(mse)
r2 = r2_score(test_data, y_pred_mean)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
