import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your real sales data from a CSV file (replace 'data/sales_data.csv' with your file path)
df = pd.read_csv('data/sales_data.csv')

# Ensure that 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Feature engineering: Extract more relevant date features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['DayOfYear'] = df['Date'].dt.dayofyear

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train_data, test_data = df[:train_size], df[train_size:]

# Prepare the input features and target variable
X_train = train_data[['Year', 'Month', 'DayOfWeek', 'DayOfYear']]
y_train = train_data['Sales']
X_test = test_data[['Year', 'Month', 'DayOfWeek', 'DayOfYear']]
y_test = test_data['Sales']

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate forecast accuracy metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Calculate forecast bias
bias = np.mean(y_pred - y_test)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
print(f"Forecast Bias: {bias:.2f}")
